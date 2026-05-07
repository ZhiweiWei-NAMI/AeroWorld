#include "AeroSemanticStencil.h"

#include "Components/PrimitiveComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "Materials/MaterialInterface.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
FString NormalizeForMatch(const FString& Value)
{
	return Value.ToLower();
}

bool ContainsPattern(const FString& Haystack, const FString& Pattern)
{
	const FString CleanPattern = NormalizeForMatch(Pattern.TrimStartAndEnd());
	if (CleanPattern.IsEmpty())
	{
		return false;
	}
	return NormalizeForMatch(Haystack).Contains(CleanPattern);
}

bool AnyPatternMatches(const FString& Haystack, const TArray<FString>& Patterns)
{
	if (Patterns.Num() <= 0)
	{
		return true;
	}
	for (const FString& Pattern : Patterns)
	{
		if (ContainsPattern(Haystack, Pattern))
		{
			return true;
		}
	}
	return false;
}

bool AnyValueMatches(const TArray<FString>& Values, const TArray<FString>& Patterns)
{
	if (Patterns.Num() <= 0)
	{
		return true;
	}
	for (const FString& Value : Values)
	{
		for (const FString& Pattern : Patterns)
		{
			if (ContainsPattern(Value, Pattern))
			{
				return true;
			}
		}
	}
	return false;
}

void ReadStringArrayField(const TSharedPtr<FJsonObject>& Object, const TCHAR* FieldName, TArray<FString>& OutValues)
{
	if (!Object.IsValid() || !Object->HasField(FieldName))
	{
		return;
	}

	FString SingleValue;
	if (Object->TryGetStringField(FieldName, SingleValue))
	{
		if (!SingleValue.TrimStartAndEnd().IsEmpty())
		{
			OutValues.Add(SingleValue.TrimStartAndEnd());
		}
		return;
	}

	if (!Object->HasTypedField<EJson::Array>(FieldName))
	{
		return;
	}

	for (const TSharedPtr<FJsonValue>& Value : Object->GetArrayField(FieldName))
	{
		if (!Value.IsValid())
		{
			continue;
		}
		const FString Text = Value->AsString().TrimStartAndEnd();
		if (!Text.IsEmpty())
		{
			OutValues.Add(Text);
		}
	}
}

TArray<TSharedPtr<FJsonValue>> StringArrayToJson(const TArray<FString>& Values)
{
	TArray<TSharedPtr<FJsonValue>> Result;
	for (const FString& Value : Values)
	{
		Result.Add(MakeShared<FJsonValueString>(Value));
	}
	return Result;
}

TSharedPtr<FJsonObject> ClassIdToNameJson(const TMap<uint8, FString>& Values)
{
	TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
	TArray<uint8> Keys;
	Values.GetKeys(Keys);
	Keys.Sort();
	for (const uint8 Key : Keys)
	{
		Object->SetStringField(FString::FromInt(Key), Values[Key]);
	}
	return Object;
}

TSharedPtr<FJsonObject> ClassNameToIdJson(const TMap<FString, uint8>& Values)
{
	TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
	TArray<FString> Keys;
	Values.GetKeys(Keys);
	Keys.Sort();
	for (const FString& Key : Keys)
	{
		Object->SetNumberField(Key, Values[Key]);
	}
	return Object;
}

TSharedPtr<FJsonObject> HistogramJson(const TMap<uint8, int32>& Values)
{
	TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
	TArray<uint8> Keys;
	Values.GetKeys(Keys);
	Keys.Sort();
	for (const uint8 Key : Keys)
	{
		Object->SetNumberField(FString::FromInt(Key), Values[Key]);
	}
	return Object;
}

FString ActorLabel(const AActor* Actor)
{
#if WITH_EDITOR
	return Actor != nullptr ? Actor->GetActorLabel() : FString();
#else
	return FString();
#endif
}

FString ActorFolderPath(const AActor* Actor)
{
#if WITH_EDITOR
	return Actor != nullptr ? Actor->GetFolderPath().ToString() : FString();
#else
	return FString();
#endif
}

TArray<FString> ActorTags(const AActor* Actor, const UPrimitiveComponent* Component)
{
	TArray<FString> Values;
	if (Actor != nullptr)
	{
		for (const FName& Tag : Actor->Tags)
		{
			Values.Add(Tag.ToString());
		}
	}
	if (Component != nullptr)
	{
		for (const FName& Tag : Component->ComponentTags)
		{
			Values.Add(Tag.ToString());
		}
	}
	return Values;
}

TArray<FString> ComponentMaterials(const UPrimitiveComponent* Component)
{
	TArray<FString> Values;
	if (Component == nullptr)
	{
		return Values;
	}

	const int32 MaterialCount = Component->GetNumMaterials();
	for (int32 Index = 0; Index < MaterialCount; ++Index)
	{
		const UMaterialInterface* Material = Component->GetMaterial(Index);
		if (Material == nullptr)
		{
			continue;
		}
		Values.Add(Material->GetName());
		Values.Add(Material->GetPathName());
	}
	return Values;
}

FString CombinedKey(
	const AActor* Actor,
	const UPrimitiveComponent* Component,
	const FString& InActorLabel,
	const FString& FolderPath,
	const TArray<FString>& Tags,
	const TArray<FString>& Materials)
{
	FString Key;
	if (Actor != nullptr)
	{
		Key += Actor->GetName();
		Key += TEXT(" ");
		Key += InActorLabel;
		Key += TEXT(" ");
		Key += (Actor->GetClass() != nullptr ? Actor->GetClass()->GetName() : FString());
	}
	if (Component != nullptr)
	{
		Key += TEXT(" ");
		Key += Component->GetName();
		Key += TEXT(" ");
		Key += (Component->GetClass() != nullptr ? Component->GetClass()->GetName() : FString());
	}
	Key += TEXT(" ");
	Key += FolderPath;
	for (const FString& Tag : Tags)
	{
		Key += TEXT(" ");
		Key += Tag;
	}
	for (const FString& Material : Materials)
	{
		Key += TEXT(" ");
		Key += Material;
	}
	return Key;
}

const FAeroSemanticStencilRule* MatchRule(
	const AActor* Actor,
	const UPrimitiveComponent* Component,
	const TArray<FAeroSemanticStencilRule>& Rules,
	const FString& InActorLabel,
	const FString& FolderPath,
	const TArray<FString>& Tags,
	const TArray<FString>& Materials)
{
	const FString ActorName = Actor != nullptr ? Actor->GetName() : FString();
	const FString ActorClass = Actor != nullptr && Actor->GetClass() != nullptr ? Actor->GetClass()->GetName() : FString();
	const FString ComponentName = Component != nullptr ? Component->GetName() : FString();
	const FString ComponentClass = Component != nullptr && Component->GetClass() != nullptr ? Component->GetClass()->GetName() : FString();
	const FString Key = CombinedKey(Actor, Component, InActorLabel, FolderPath, Tags, Materials);

	for (const FAeroSemanticStencilRule& Rule : Rules)
	{
		if (!AnyPatternMatches(Key, Rule.Patterns))
		{
			continue;
		}
		if (!AnyPatternMatches(ActorName + TEXT(" ") + InActorLabel, Rule.ActorPatterns))
		{
			continue;
		}
		if (!AnyPatternMatches(ComponentName, Rule.ComponentPatterns))
		{
			continue;
		}
		if (!AnyPatternMatches(ActorClass, Rule.ActorClassPatterns))
		{
			continue;
		}
		if (!AnyPatternMatches(ComponentClass, Rule.ComponentClassPatterns))
		{
			continue;
		}
		if (!AnyValueMatches(Tags, Rule.TagPatterns))
		{
			continue;
		}
		if (!AnyValueMatches(Materials, Rule.MaterialPatterns))
		{
			continue;
		}
		if (!AnyPatternMatches(FolderPath, Rule.FolderPatterns))
		{
			continue;
		}
		return &Rule;
	}
	return nullptr;
}

void AddDefaultClasses(TMap<FString, uint8>& ClassNameToId, TMap<uint8, FString>& ClassIdToName)
{
	struct FDefaultClass
	{
		const TCHAR* Name;
		uint8 Id;
	};
	const FDefaultClass Defaults[] = {
		{TEXT("ignore"), 0},
		{TEXT("city_base_background"), 1},
		{TEXT("building"), 2},
		{TEXT("vegetation"), 3},
		{TEXT("water"), 4},
		{TEXT("vehicle"), 5},
		{TEXT("pedestrian"), 6},
		{TEXT("drone"), 7},
		{TEXT("obstacle"), 8},
		{TEXT("traffic_control"), 9},
		{TEXT("facility"), 10},
		{TEXT("hazard_trigger"), 11},
	};
	for (const FDefaultClass& Entry : Defaults)
	{
		ClassNameToId.FindOrAdd(Entry.Name, Entry.Id);
		ClassIdToName.FindOrAdd(Entry.Id, Entry.Name);
	}
}

bool ResolveRuleClass(
	const TSharedPtr<FJsonObject>& RuleObject,
	const TMap<FString, uint8>& ClassNameToId,
	uint8& OutClassId,
	FString& OutClassName,
	FString& OutError)
{
	double ClassIdNumber = -1.0;
	if (RuleObject->TryGetNumberField(TEXT("class_id"), ClassIdNumber))
	{
		const int32 ClassIdInt = FMath::Clamp(FMath::RoundToInt(ClassIdNumber), 0, 255);
		OutClassId = static_cast<uint8>(ClassIdInt);
		OutClassName = FString::Printf(TEXT("class_%d"), ClassIdInt);
		for (const TPair<FString, uint8>& Pair : ClassNameToId)
		{
			if (Pair.Value == OutClassId)
			{
				OutClassName = Pair.Key;
				break;
			}
		}
		return true;
	}

	FString ClassName;
	if (!RuleObject->TryGetStringField(TEXT("class"), ClassName))
	{
		RuleObject->TryGetStringField(TEXT("class_name"), ClassName);
	}
	ClassName = ClassName.TrimStartAndEnd();
	const uint8* ClassId = ClassNameToId.Find(ClassName);
	if (ClassId == nullptr)
	{
		OutError = FString::Printf(TEXT("semantic rule references unknown class '%s'."), *ClassName);
		return false;
	}
	OutClassId = *ClassId;
	OutClassName = ClassName;
	return true;
}
}

FString AeroSemanticStencil::DefaultRulesPath()
{
	return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("Config/LowAltitude/semantic_stencil_rules.json"));
}

bool AeroSemanticStencil::LoadRules(
	const FString& RulesPath,
	TArray<FAeroSemanticStencilRule>& OutRules,
	TMap<FString, uint8>& OutClassNameToId,
	TMap<uint8, FString>& OutClassIdToName,
	FString& OutError)
{
	OutRules.Reset();
	OutClassNameToId.Reset();
	OutClassIdToName.Reset();
	AddDefaultClasses(OutClassNameToId, OutClassIdToName);

	const FString ResolvedPath = RulesPath.TrimStartAndEnd().IsEmpty() ? DefaultRulesPath() : RulesPath;
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *ResolvedPath))
	{
		OutError = FString::Printf(TEXT("failed to read semantic stencil rules: %s"), *ResolvedPath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		OutError = FString::Printf(TEXT("failed to parse semantic stencil rules: %s"), *ResolvedPath);
		return false;
	}

	if (Root->HasTypedField<EJson::Object>(TEXT("classes")))
	{
		const TSharedPtr<FJsonObject> Classes = Root->GetObjectField(TEXT("classes"));
		for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Classes->Values)
		{
			const int32 ClassId = FMath::Clamp(FMath::RoundToInt(Pair.Value->AsNumber()), 0, 255);
			OutClassNameToId.Add(Pair.Key, static_cast<uint8>(ClassId));
			OutClassIdToName.Add(static_cast<uint8>(ClassId), Pair.Key);
		}
	}

	if (!Root->HasTypedField<EJson::Array>(TEXT("rules")))
	{
		OutError = FString::Printf(TEXT("semantic stencil rules file has no rules array: %s"), *ResolvedPath);
		return false;
	}

	int32 Order = 0;
	for (const TSharedPtr<FJsonValue>& RuleValue : Root->GetArrayField(TEXT("rules")))
	{
		const TSharedPtr<FJsonObject> RuleObject = RuleValue.IsValid() ? RuleValue->AsObject() : nullptr;
		if (!RuleObject.IsValid())
		{
			continue;
		}

		FAeroSemanticStencilRule Rule;
		Rule.Order = Order++;
		if (!ResolveRuleClass(RuleObject, OutClassNameToId, Rule.ClassId, Rule.ClassName, OutError))
		{
			return false;
		}
		double PriorityNumber = 0.0;
		if (RuleObject->TryGetNumberField(TEXT("priority"), PriorityNumber))
		{
			Rule.Priority = FMath::RoundToInt(PriorityNumber);
		}
		ReadStringArrayField(RuleObject, TEXT("pattern"), Rule.Patterns);
		ReadStringArrayField(RuleObject, TEXT("patterns"), Rule.Patterns);
		ReadStringArrayField(RuleObject, TEXT("actor"), Rule.ActorPatterns);
		ReadStringArrayField(RuleObject, TEXT("actor_patterns"), Rule.ActorPatterns);
		ReadStringArrayField(RuleObject, TEXT("component"), Rule.ComponentPatterns);
		ReadStringArrayField(RuleObject, TEXT("component_patterns"), Rule.ComponentPatterns);
		ReadStringArrayField(RuleObject, TEXT("actor_class"), Rule.ActorClassPatterns);
		ReadStringArrayField(RuleObject, TEXT("actor_class_patterns"), Rule.ActorClassPatterns);
		ReadStringArrayField(RuleObject, TEXT("component_class"), Rule.ComponentClassPatterns);
		ReadStringArrayField(RuleObject, TEXT("component_class_patterns"), Rule.ComponentClassPatterns);
		ReadStringArrayField(RuleObject, TEXT("tag"), Rule.TagPatterns);
		ReadStringArrayField(RuleObject, TEXT("tag_patterns"), Rule.TagPatterns);
		ReadStringArrayField(RuleObject, TEXT("material"), Rule.MaterialPatterns);
		ReadStringArrayField(RuleObject, TEXT("material_patterns"), Rule.MaterialPatterns);
		ReadStringArrayField(RuleObject, TEXT("folder"), Rule.FolderPatterns);
		ReadStringArrayField(RuleObject, TEXT("folder_patterns"), Rule.FolderPatterns);
		OutRules.Add(Rule);
	}

	OutRules.Sort(
		[](const FAeroSemanticStencilRule& A, const FAeroSemanticStencilRule& B)
		{
			if (A.Priority != B.Priority)
			{
				return A.Priority > B.Priority;
			}
			return A.Order < B.Order;
		});
	return true;
}

bool AeroSemanticStencil::AuditAndAssign(
	UWorld* World,
	const FString& RulesPath,
	bool bAssign,
	const TSet<const AActor*>& IgnoredActors,
	FAeroSemanticStencilAudit& OutAudit,
	FString& OutError)
{
	OutAudit = FAeroSemanticStencilAudit();
	OutAudit.RulesPath = RulesPath.TrimStartAndEnd().IsEmpty() ? DefaultRulesPath() : RulesPath;
	OutAudit.bAssigned = bAssign;

	TArray<FAeroSemanticStencilRule> Rules;
	if (!LoadRules(OutAudit.RulesPath, Rules, OutAudit.ClassNameToId, OutAudit.ClassIdToName, OutError))
	{
		return false;
	}

	if (World == nullptr)
	{
		OutError = TEXT("semantic stencil audit requires a world.");
		return false;
	}

	for (TActorIterator<AActor> It(World); It; ++It)
	{
		AActor* Actor = *It;
		if (!IsValid(Actor) || Actor->IsHidden() || IgnoredActors.Contains(Actor))
		{
			continue;
		}
		++OutAudit.ActorCount;

		const FString Label = ActorLabel(Actor);
		const FString FolderPath = ActorFolderPath(Actor);
		TArray<UPrimitiveComponent*> Components;
		Actor->GetComponents<UPrimitiveComponent>(Components);
		for (UPrimitiveComponent* Component : Components)
		{
			if (!IsValid(Component))
			{
				continue;
			}
			++OutAudit.PrimitiveComponentCount;

			FAeroSemanticStencilComponentAudit Row;
			Row.ActorName = Actor->GetName();
			Row.ActorLabel = Label;
			Row.ActorClass = Actor->GetClass() != nullptr ? Actor->GetClass()->GetName() : FString();
			Row.ComponentName = Component->GetName();
			Row.ComponentClass = Component->GetClass() != nullptr ? Component->GetClass()->GetName() : FString();
			Row.FolderPath = FolderPath;
			Row.Tags = ActorTags(Actor, Component);
			Row.Materials = ComponentMaterials(Component);
			Row.bVisible = Component->IsVisible();
			Row.bRegistered = Component->IsRegistered();
			Row.bRenderCustomDepthBefore = Component->bRenderCustomDepth;
			Row.StencilBefore = Component->CustomDepthStencilValue;
			Row.StencilAfter = Row.StencilBefore;
			Row.Bounds = Component->Bounds.GetBox().ToString();

			const FAeroSemanticStencilRule* MatchedRule = MatchRule(Actor, Component, Rules, Label, FolderPath, Row.Tags, Row.Materials);
			if (MatchedRule != nullptr)
			{
				Row.MatchedClassId = MatchedRule->ClassId;
				Row.MatchedClassName = MatchedRule->ClassName;
				Row.MatchedRulePattern = MatchedRule->Patterns.Num() > 0 ? MatchedRule->Patterns[0] : FString();
			}
			else
			{
				Row.MatchedClassId = 0;
				Row.MatchedClassName = OutAudit.ClassIdToName.FindRef(0);
				if (Row.MatchedClassName.IsEmpty())
				{
					Row.MatchedClassName = TEXT("ignore");
				}
			}

			if (bAssign && Row.bVisible && Row.bRegistered)
			{
				Component->SetRenderCustomDepth(true);
				Component->SetCustomDepthStencilValue(static_cast<int32>(Row.MatchedClassId));
				Component->MarkRenderStateDirty();
				Row.StencilAfter = Component->CustomDepthStencilValue;
				++OutAudit.AssignedComponentCount;
				OutAudit.AssignedComponentHistogram.FindOrAdd(Row.MatchedClassId) += 1;
			}

			OutAudit.Components.Add(MoveTemp(Row));
		}
	}

	return true;
}

TSharedPtr<FJsonObject> AeroSemanticStencil::AuditToJson(const FAeroSemanticStencilAudit& Audit, bool bIncludeComponents)
{
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetStringField(TEXT("schema"), TEXT("aeroworld_semantic_stencil_audit_v1"));
	Root->SetStringField(TEXT("rules_path"), Audit.RulesPath);
	Root->SetBoolField(TEXT("assigned"), Audit.bAssigned);
	Root->SetNumberField(TEXT("actor_count"), Audit.ActorCount);
	Root->SetNumberField(TEXT("primitive_component_count"), Audit.PrimitiveComponentCount);
	Root->SetNumberField(TEXT("assigned_component_count"), Audit.AssignedComponentCount);
	Root->SetObjectField(TEXT("class_name_to_id"), ClassNameToIdJson(Audit.ClassNameToId));
	Root->SetObjectField(TEXT("semantic_class_by_id"), ClassIdToNameJson(Audit.ClassIdToName));
	Root->SetObjectField(TEXT("assigned_component_histogram"), HistogramJson(Audit.AssignedComponentHistogram));

	if (bIncludeComponents)
	{
		TArray<TSharedPtr<FJsonValue>> ComponentValues;
		for (const FAeroSemanticStencilComponentAudit& Row : Audit.Components)
		{
			TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
			Object->SetStringField(TEXT("actor"), Row.ActorName);
			Object->SetStringField(TEXT("actor_label"), Row.ActorLabel);
			Object->SetStringField(TEXT("actor_class"), Row.ActorClass);
			Object->SetStringField(TEXT("component"), Row.ComponentName);
			Object->SetStringField(TEXT("component_class"), Row.ComponentClass);
			Object->SetStringField(TEXT("folder_path"), Row.FolderPath);
			Object->SetArrayField(TEXT("tags"), StringArrayToJson(Row.Tags));
			Object->SetArrayField(TEXT("materials"), StringArrayToJson(Row.Materials));
			Object->SetBoolField(TEXT("visible"), Row.bVisible);
			Object->SetBoolField(TEXT("registered"), Row.bRegistered);
			Object->SetBoolField(TEXT("render_custom_depth_before"), Row.bRenderCustomDepthBefore);
			Object->SetNumberField(TEXT("current_stencil"), Row.StencilBefore);
			Object->SetNumberField(TEXT("assigned_stencil"), Row.StencilAfter);
			Object->SetNumberField(TEXT("matched_class_id"), Row.MatchedClassId);
			Object->SetStringField(TEXT("matched_class"), Row.MatchedClassName);
			Object->SetStringField(TEXT("matched_rule_pattern"), Row.MatchedRulePattern);
			Object->SetStringField(TEXT("bounds"), Row.Bounds);
			ComponentValues.Add(MakeShared<FJsonValueObject>(Object));
		}
		Root->SetArrayField(TEXT("components"), ComponentValues);
	}

	return Root;
}

bool AeroSemanticStencil::SaveAuditJson(const FAeroSemanticStencilAudit& Audit, const FString& AuditPath, FString& OutError)
{
	if (AuditPath.TrimStartAndEnd().IsEmpty())
	{
		return true;
	}

	const FString Directory = FPaths::GetPath(AuditPath);
	if (!Directory.IsEmpty() && !IFileManager::Get().MakeDirectory(*Directory, true))
	{
		OutError = FString::Printf(TEXT("failed to create semantic audit directory: %s"), *Directory);
		return false;
	}

	FString Output;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	FJsonSerializer::Serialize(AuditToJson(Audit, true).ToSharedRef(), Writer);
	if (!FFileHelper::SaveStringToFile(Output, *AuditPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
	{
		OutError = FString::Printf(TEXT("failed to save semantic stencil audit: %s"), *AuditPath);
		return false;
	}
	return true;
}
